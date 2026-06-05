"""Tests for Exp 3863 graph-verifier facts complementarity.

Spec: REQ-VERIFY-3863, SCENARIO-VERIFY-3863.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from carnot.verify import graph_verifier_facts_complementarity_v2 as mod


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "openspec" / "capabilities" / "verification" / "spec.md"


def _score_rows() -> list[dict[str, Any]]:
    return [
        {
            "item_id": "g-catches-0",
            "gold_ungrounded": True,
            "graph_score": 0.9,
            "math_ensemble_score": 0.2,
        },
        {
            "item_id": "both-catch",
            "gold_ungrounded": True,
            "graph_score": 0.8,
            "math_ensemble_score": 0.8,
        },
        {
            "item_id": "ensemble-catches",
            "gold_ungrounded": True,
            "graph_score": 0.2,
            "math_ensemble_score": 0.9,
        },
        {
            "item_id": "g-catches-1",
            "gold_ungrounded": True,
            "graph_score": 0.9,
            "math_ensemble_score": 0.2,
        },
        {
            "item_id": "both-grounded-correct",
            "gold_ungrounded": False,
            "graph_score": 0.1,
            "math_ensemble_score": 0.1,
        },
        {
            "item_id": "graph-false-positive",
            "gold_ungrounded": False,
            "graph_score": 0.8,
            "math_ensemble_score": 0.1,
        },
        {
            "item_id": "ensemble-false-positive",
            "gold_ungrounded": False,
            "graph_score": 0.1,
            "math_ensemble_score": 0.8,
        },
    ]


def _upstream_payload(
    *,
    facts_catch_delta: float = 0.2,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "honest_verdict": "complete: graph_grounding_prototype_SIGNAL_fixture",
        "facts_catch_delta": facts_catch_delta,
        "n_facts_items": len(rows or _score_rows()),
        "per_item_scores": rows if rows is not None else _score_rows(),
    }


def _write_upstream(root: Path, payload: dict[str, Any]) -> Path:
    output = root / mod.UPSTREAM_REL_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def test_req_verify_3863_spec_anchor_exists() -> None:
    """REQ-VERIFY-3863: the complementarity artifact is OpenSpec anchored."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-VERIFY-3863" in spec
    assert "SCENARIO-VERIFY-3863" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "facts_error_mask_correlation" in spec


def test_scenario_verify_3863_computes_independent_catches_and_phi() -> None:
    """SCENARIO-VERIFY-3863: cached masks produce complementarity metrics."""

    items = mod.parse_per_item_scores(_upstream_payload())
    metrics = mod.compute_complementarity_metrics(items)

    assert metrics.graph_catches_ensemble_misses == 2
    assert metrics.ensemble_catches_graph_misses == 1
    assert metrics.graph_catch_rate == pytest.approx(0.75)
    assert metrics.ensemble_catch_rate == pytest.approx(0.5)
    assert metrics.union_facts_catch_rate == pytest.approx(1.0)
    assert metrics.union_lift_over_ensemble == pytest.approx(0.5)
    assert metrics.facts_error_mask_correlation == pytest.approx(-0.5477225575)
    assert metrics.extended_ensemble_recommended is True
    assert metrics.error_mask_confusion == {"both_error": 0, "graph_only": 2, "ensemble_only": 3, "both_correct": 2}


def test_req_verify_3863_artifact_includes_principles_and_upstream_sha(tmp_path: Path) -> None:
    """REQ-VERIFY-3863: artifact fields, principles, and SHA citation are stable."""

    upstream_path = _write_upstream(tmp_path, _upstream_payload())
    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.5)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: graph_verifier_COMPLEMENTARY_catches2_corr-0.548")
    assert artifact["graph_catches_ensemble_misses"] == 2
    assert artifact["ensemble_catches_graph_misses"] == 1
    assert artifact["facts_error_mask_correlation"] == pytest.approx(-0.547723)
    assert artifact["union_facts_catch_rate"] == pytest.approx(1.0)
    assert artifact["extended_ensemble_recommended"] is True
    assert artifact["n_facts_items"] == 7
    assert artifact["duration_s"] == 2.5
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["cited_upstream_artifacts"] == [
        {
            "experiment_id": 3862,
            "path": str(upstream_path.relative_to(tmp_path)),
            "sha256": mod.sha256_file(upstream_path),
            "facts_catch_delta": 0.2,
            "n_facts_items": 7,
        }
    ]
    assert set(mod.REQUIRED_PRINCIPLE_FIELDS) <= set(artifact["field_principles"])
    assert all(
        "principle" in artifact["field_principles"][field]
        for field in mod.REQUIRED_PRINCIPLE_FIELDS
    )


def test_scenario_verify_3863_redundant_verdict_for_low_independent_catch(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3863: low independent graph catch blocks recommendation."""

    rows = [
        {"item_id": "a", "gold_ungrounded": True, "graph_score": 0.9, "math_ensemble_score": 0.9},
        {"item_id": "b", "gold_ungrounded": True, "graph_score": 0.1, "math_ensemble_score": 0.1},
        {"item_id": "c", "gold_ungrounded": False, "graph_score": 0.9, "math_ensemble_score": 0.9},
        {"item_id": "d", "gold_ungrounded": False, "graph_score": 0.1, "math_ensemble_score": 0.1},
    ]
    _write_upstream(tmp_path, _upstream_payload(rows=rows))

    artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    mod.validate_artifact(artifact)
    assert artifact["extended_ensemble_recommended"] is False
    assert artifact["graph_catches_ensemble_misses"] == 0
    assert artifact["facts_error_mask_correlation"] == pytest.approx(1.0)
    assert artifact["honest_verdict"].startswith(
        "complete: graph_verifier_REDUNDANT_with_math_ensemble_on_facts_low_independent_catch0"
    )


@pytest.mark.parametrize(
    ("payload", "expected_detail"),
    [
        (None, "missing"),
        (_upstream_payload(facts_catch_delta=0.0), "facts_catch_delta"),
        ({"facts_catch_delta": 0.2, "n_facts_items": 7}, "per-item"),
        (
            {
                "facts_catch_delta": 0.2,
                "per_item_scores": [
                    {"item_id": "grounded", "gold_ungrounded": False, "graph_score": 0.1, "math_ensemble_score": 0.1}
                ],
            },
            "gold_ungrounded",
        ),
    ],
)
def test_scenario_verify_3863_blocked_graph_prototype_unavailable(
    tmp_path: Path,
    payload: dict[str, Any] | None,
    expected_detail: str,
) -> None:
    """SCENARIO-VERIFY-3863: failed preconditions do not fabricate metrics."""

    if payload is not None:
        _write_upstream(tmp_path, payload)

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=1.25)

    mod.validate_artifact(artifact)
    assert artifact["honest_verdict"] == "blocked_graph_prototype_unavailable"
    assert artifact["extended_ensemble_recommended"] is False
    assert artifact["graph_catches_ensemble_misses"] == 0
    assert artifact["facts_error_mask_correlation"] is None
    assert artifact["n_facts_items"] == 0
    assert expected_detail in artifact["blocked_detail"]


def test_req_verify_3863_write_artifact_and_cli_persist_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3863: writer and experiment script persist the terminal JSON."""

    _write_upstream(tmp_path, _upstream_payload())
    output = mod.write_artifact(tmp_path)

    saved = json.loads(output.read_text(encoding="utf-8"))
    mod.validate_artifact(saved)
    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["honest_verdict"].startswith("complete:")

    script = REPO_ROOT / "scripts" / "experiments" / "experiment_3863_graph_verifier_facts_complementarity_v2.py"
    result = subprocess.run(
        [sys.executable, str(script), "--repo-root", str(tmp_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert str(tmp_path / mod.OUTPUT_REL_PATH) in result.stdout
    mod.validate_artifact(json.loads((tmp_path / mod.OUTPUT_REL_PATH).read_text(encoding="utf-8")))


def test_req_verify_3863_validation_and_parser_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3863: schema validation and item parsing fail closed."""

    payload = _upstream_payload(
        rows=[
            {"id": "nested", "is_hallucination": 1, "scores": {"graph_grounding": 0.7, "math_ensemble": 0.2}},
            {"id": "bad", "is_hallucination": "maybe", "graph_score": 0.7, "math_ensemble_score": 0.2},
            {"id": "nan", "is_hallucination": 1, "graph_score": "nan", "math_ensemble_score": 0.2},
        ]
    )
    items = mod.parse_per_item_scores(payload)

    assert items == (
        mod.FactsScoreItem(
            item_id="nested",
            gold_ungrounded=True,
            graph_score=0.7,
            ensemble_score=0.2,
        ),
    )
    assert mod.matthews_phi([0, 0], [1, 1]) == 0.0
    assert mod.relative_path(tmp_path, tmp_path / "x.json") == "x.json"
    assert mod.relative_path(tmp_path, Path("/outside/x.json")) == "/outside/x.json"

    artifact = mod.build_blocked_artifact(
        reason="blocked_graph_prototype_unavailable",
        blocked_detail="fixture",
        preconditions_checked=[],
        cited_upstream_artifacts=[],
        started_s=0.0,
        finished_s=1.0,
    )
    broken = dict(artifact)
    broken.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["honest_verdict"] = "pending"
    with pytest.raises(ValueError, match="terminal prefix"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["field_principles"] = []
    with pytest.raises(ValueError, match="field_principles"):
        mod.validate_artifact(broken)

    broken = dict(artifact)
    broken["extended_ensemble_recommended"] = 1
    with pytest.raises(ValueError, match="bare bool"):
        mod.validate_artifact(broken)

    _write_upstream(tmp_path, payload)
    blocked = mod.build_artifact(tmp_path, started_s=2.0, now_s=3.0)
    assert blocked["honest_verdict"] == "blocked_graph_prototype_unavailable"
    assert "gold_ungrounded" in blocked["blocked_detail"]


def test_req_verify_3863_defensive_branches_are_covered(tmp_path: Path) -> None:
    """REQ-VERIFY-3863: defensive parsing and validation branches stay deterministic."""

    malformed = tmp_path / mod.UPSTREAM_REL_PATH
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{bad-json", encoding="utf-8")
    malformed_artifact = mod.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert malformed_artifact["honest_verdict"] == "blocked_graph_prototype_unavailable"
    assert "malformed" in malformed_artifact["blocked_detail"]
    assert malformed_artifact["cited_upstream_artifacts"][0]["facts_catch_delta"] is None

    assert mod.sha256_file(tmp_path / "missing.json") is None
    with pytest.raises(ValueError, match="same length"):
        mod.matthews_phi([1], [1, 0])

    high_corr = mod.ComplementarityMetrics(
        n_facts_items=4,
        n_gold_ungrounded=2,
        graph_catches_ensemble_misses=1,
        ensemble_catches_graph_misses=0,
        graph_catch_rate=1.0,
        ensemble_catch_rate=0.5,
        union_facts_catch_rate=1.0,
        union_lift_over_ensemble=0.5,
        facts_error_mask_correlation=0.9,
        extended_ensemble_recommended=False,
        error_mask_confusion={"both_error": 1, "graph_only": 0, "ensemble_only": 0, "both_correct": 3},
        graph_catches_ensemble_miss_ids=("x",),
        ensemble_catches_graph_miss_ids=(),
    )
    boundary = mod.ComplementarityMetrics(
        **{
            **high_corr.__dict__,
            "facts_error_mask_correlation": 0.1,
            "extended_ensemble_recommended": False,
        }
    )
    assert mod.classify_verdict(high_corr).endswith("high_corr0.900")
    assert mod.classify_verdict(boundary).endswith("boundary")

    nested_payload = {
        "outer": {
            "per_item_scores": [
                {
                    "row_id": "graph-nested",
                    "label": "incorrect",
                    "graph": {"energy": 0.8},
                    "ensemble": {"score": 0.7},
                },
                {
                    "row_id": "ensemble-nested",
                    "label": "correct",
                    "graph": {"score": 0.1},
                    "math_ensemble": {"energy": 0.2},
                },
                {"row_id": "missing-score", "label": "yes"},
                {"row_id": "bad-score", "label": "yes", "graph_score": "bad", "math_ensemble_score": 0.2},
                {"row_id": "bad-nested", "label": "yes", "graph": {"x": 1}, "ensemble": {"x": 1}},
            ]
        }
    }
    assert mod.parse_per_item_scores(nested_payload) == (
        mod.FactsScoreItem("graph-nested", True, 0.8, 0.7),
        mod.FactsScoreItem("ensemble-nested", False, 0.1, 0.2),
    )

    _write_upstream(tmp_path, _upstream_payload())
    complete = mod.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    mutations: list[tuple[str, Any, str]] = [
        ("field_principles", {"honest_verdict": "principle: x"}, "field_principles missing"),
        (
            "field_principles",
            {**complete["field_principles"], "duration_s": "missing note"},
            "principle note",
        ),
        ("n_facts_items", -1, "n_facts_items"),
        ("preconditions_checked", {}, "preconditions_checked"),
        ("cited_upstream_artifacts", {}, "cited_upstream_artifacts"),
        ("random_seed", "3863", "random_seed"),
        ("reproducibility_checksum", "short", "reproducibility_checksum"),
        ("duration_s", -1.0, "duration_s"),
        ("graph_catches_ensemble_misses", "2", "graph_catches_ensemble_misses"),
        ("facts_error_mask_correlation", 2.0, "facts_error_mask_correlation"),
        ("union_facts_catch_rate", 2.0, "union_facts_catch_rate"),
    ]
    for field, value, message in mutations:
        broken = dict(complete)
        broken[field] = value
        with pytest.raises(ValueError, match=message):
            mod.validate_artifact(broken)

    blocked = mod.build_blocked_artifact(
        reason="blocked_graph_prototype_unavailable",
        blocked_detail="fixture",
        preconditions_checked=[],
        cited_upstream_artifacts=[],
        started_s=0.0,
        finished_s=1.0,
    )
    blocked["honest_verdict"] = "blocked_other"
    with pytest.raises(ValueError, match="blocked_graph_prototype_unavailable"):
        mod.validate_artifact(blocked)
