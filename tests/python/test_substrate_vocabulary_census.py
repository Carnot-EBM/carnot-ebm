"""Tests for the substrate vocabulary census.

Spec: REQ-SUBSTRATE-CENSUS-1 (openspec/capabilities/research-harnesses/spec.md).
SCENARIOs: SCENARIO-SUBSTRATE-CENSUS-1-SHAPES (every field shape is named),
SCENARIO-SUBSTRATE-CENSUS-1-GATE-VIEW (the census reports what the gate does, not a
re-derivation), SCENARIO-SUBSTRATE-CENSUS-1-READ-ONLY (the sweep never writes),
SCENARIO-SUBSTRATE-CENSUS-1-UNREADABLE (a missing directory is not "zero").

Every test runs over a corpus built under tmp_path. Nothing here reads or writes
results/**.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import substrate_vocabulary_census as census  # noqa: E402

# One artifact per shape the corpus actually contains (measured 2026-09-05).
CORPUS = {
    "experiment_1_live.json": {
        "inference_substrate": "live_llm_inference",
        "duration_s": 120.0,
        "model_specs": [{"hf_id": "unsloth/gemma-4-31B-it-GGUF"}],
    },
    "experiment_2_agg_noted.json": {
        "inference_substrate": "aggregation_from_upstream_artifacts -- reads upstream JSON",
        "duration_s": 0.01,
    },
    "experiment_3_suffix.json": {
        "inference_substrate": "deterministic_arc_live_attempt_fixture_no_llm",
        "duration_s": 0.4,
    },
    "experiment_4_wrapped.json": {
        "inference_substrate": {
            "value": "verifier_ensemble_against_cached_candidates",
            "principle": "scores cached triples",
        },
        "duration_s": 3.0,
    },
    "experiment_5_dict.json": {
        "inference_substrate": {"kind": "aggregation", "no_live_llm_inference": True},
        "duration_s": 0.2,
    },
    "experiment_6_unknown.json": {
        "inference_substrate": "live_llm_arc_belief_shadow",
        "duration_s": 0.0,
    },
    "experiment_7_missing.json": {"duration_s": 1.0},
}


def _build(tmp_path: Path) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    for name, payload in CORPUS.items():
        (results / name).write_text(json.dumps(payload), encoding="utf-8")
    (results / "experiment_8_broken.json").write_text("{not json", encoding="utf-8")
    return results


def _digest(results: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in results.iterdir()}


def test_every_field_shape_is_named() -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-SHAPES: a dict without `value` is its own shape."""
    assert census.unwrap_substrate("live_llm_inference") == ("string", "live_llm_inference")
    assert census.unwrap_substrate({"value": " x ", "principle": "p"}) == (
        "principle_wrapped",
        "x",
    )
    assert census.unwrap_substrate({"kind": "aggregation"}) == ("dict_without_value", "")
    assert census.unwrap_substrate(None) == ("missing", "")
    assert census.unwrap_substrate("   ") == ("missing", "")
    assert census.unwrap_substrate(["a"]) == ("other", "")


def test_census_counts_each_population(tmp_path: Path) -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-SHAPES: the aggregate names every population."""
    report = census.census(_build(tmp_path))
    assert report["files"] == 8
    assert report["unreadable"] == 1
    assert report["shapes"] == {
        "string": 4,
        "principle_wrapped": 1,
        "dict_without_value": 1,
        "missing": 1,
    }
    assert report["declared_string"] == 5
    assert report["distinct_raw"] == 5
    # The noted aggregation value and its bare form collapse to one leading token.
    assert report["distinct_leading"] == 5
    assert report["legal_exact_artifacts"] == 2  # live + wrapped verifier
    assert report["legal_leading_artifacts"] == 3  # plus the noted aggregation
    assert report["legal_per_value"]["aggregation_from_upstream_artifacts"] == 1
    assert report["singleton_raw"] == 5


def test_census_reports_the_gates_own_view(tmp_path: Path) -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-GATE-VIEW: classifier and floor come from the gate."""
    report = census.census(_build(tmp_path))
    assert report["classifier_source"] == {
        "top_level_inference_substrate": 3,
        "no_llm_name_suffix": 1,
        "unknown_top_level_inference_substrate": 1,
    }
    assert report["effective_class"]["model_full_generation"] == 1
    assert report["effective_class"]["aggregation"] == 1
    assert report["effective_class"]["no_model_load"] == 2
    assert report["effective_class"]["unfloored"] == 1
    assert report["unknown_values"] == ["live_llm_arc_belief_shadow"]
    assert report["unfloored_without_duration"] == 1


def test_effective_class_maps_every_known_reason() -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-GATE-VIEW: no floor reason falls through unnamed."""
    assert census.effective_class(None) == "unfloored"
    assert census.effective_class({"reason": "live_model"}) == "model_full_generation"
    assert census.effective_class({"reason": "llm_embedding_extraction"}) == (
        "model_load_no_generation"
    )
    assert census.effective_class({"reason": "brand_new_reason"}) == "other:brand_new_reason"


def test_census_never_writes(tmp_path: Path) -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-READ-ONLY: every file is byte-identical after a sweep."""
    results = _build(tmp_path)
    before = _digest(results)
    census.census(results)
    census.main(["--results-dir", str(results)])
    census.main(["--results-dir", str(results), "--json"])
    assert _digest(results) == before


def test_unreadable_directory_is_not_zero(tmp_path: Path, capsys) -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-UNREADABLE: exit 2, never a clean empty report."""
    rc = census.main(["--results-dir", str(tmp_path / "does_not_exist")])
    assert rc == census.EXIT_UNREADABLE_DIR
    assert "cannot read" in capsys.readouterr().err


def test_main_renders_and_exits_zero(tmp_path: Path, capsys) -> None:
    """SCENARIO-SUBSTRATE-CENSUS-1-GATE-VIEW: the text report carries the counts."""
    rc = census.main(["--results-dir", str(_build(tmp_path)), "--top", "3"])
    out = capsys.readouterr().out
    assert rc == census.EXIT_OK
    assert "distinct_raw=5" in out
    assert "unknown_distinct=1" in out
    rc = census.main(["--results-dir", str(tmp_path / "results"), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distinct_leading"] == 5
