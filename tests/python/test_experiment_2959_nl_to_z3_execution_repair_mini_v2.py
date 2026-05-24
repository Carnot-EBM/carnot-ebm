"""Tests for Exp 2959 NL-to-Z3 execution repair mini benchmark.

Spec: REQ-BENCH-2959, SCENARIO-BENCH-2959.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.eval import llmeval_logic_z3_mini as exp2931
from carnot.eval import nl_to_z3_execution_repair_mini_v2 as exp


MANDATED = "unsloth/gemma-4-26B-A4B-it-GGUF"


def _spec() -> dict[str, Any]:
    return {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": MANDATED,
        "gpu": 0,
        "model_path": "/tmp/gemma.gguf",
    }


def _strict_payload(item: exp2931.LogicItem, answer: str | None = None) -> str:
    return json.dumps(
        {"formalization": item.formalization, "answer": answer or item.gold_answer},
        sort_keys=True,
    )


def _raw_row(item: exp2931.LogicItem, text: str) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "model_hf_id": MANDATED,
        "model_name": "Gemma4-26B-A4B-it",
        "model_path": "/tmp/gemma.gguf",
        "gpu_index": 0,
        "prompt_hash": exp2931.sha256_text(item.prompt),
        "per_item_seed": exp.RANDOM_SEED,
        "generation_source": "live_sota_llamacpp_logic_json",
        "output_text": text,
        "raw_response_path": f"/tmp/{item.item_id}.txt",
        "elapsed_seconds": 0.1,
        "blocker": None,
    }


def test_req_bench_2959_spec_anchor_exists() -> None:
    """REQ-BENCH-2959: the repair mini benchmark is OpenSpec anchored."""

    spec = Path("openspec/capabilities/benchmarks/spec.md").read_text(encoding="utf-8")

    assert "REQ-BENCH-2959" in spec
    assert "SCENARIO-BENCH-2959" in spec
    assert exp.OUTPUT_FILENAME in spec
    assert 'inference_substrate="live_llm_inference"' in spec


def test_scenario_bench_2959_repairs_fragments_and_splits_failure_categories() -> None:
    """SCENARIO-BENCH-2959: fragments are repaired, then Z3 decides acceptance."""

    items = exp.selected_logic_items(max_items=8)[:5]
    wrong_formula = {
        "facts": [],
        "rules": [],
        "exclusions": [],
        "query": items[2].formalization["query"],
    }
    z3_exception = {
        "facts": [["A", "x"]],
        "rules": [],
        "exclusions": [],
        "query": ["A"],
    }
    rows = [
        exp.evaluate_repaired_output(items[0], _strict_payload(items[0]), generation_metadata={}),
        exp.evaluate_repaired_output(
            items[1],
            f'formalization: {json.dumps(items[1].formalization, sort_keys=True)}\n'
            'formal-logic-based-answer: "necessary"',
            generation_metadata={},
        ),
        exp.evaluate_repaired_output(
            items[2],
            json.dumps({"formalization": wrong_formula, "answer": "possible"}, sort_keys=True),
            generation_metadata={},
        ),
        exp.evaluate_repaired_output(
            items[3],
            json.dumps({"formalization": z3_exception, "answer": items[3].gold_answer}),
            generation_metadata={},
        ),
        exp.evaluate_repaired_output(items[4], "not json", generation_metadata={}),
    ]
    metrics = exp.aggregate_repair_results(rows)

    assert rows[0]["failure_category"] == "solver_verified_correct"
    assert rows[1]["parse_repair_applied"] is True
    assert rows[1]["solver_answer"] == items[1].gold_answer
    assert rows[1]["failure_category"] == "wrong_answer"
    assert rows[2]["solver_answer"] == "possible"
    assert rows[2]["failure_category"] == "wrong_formula"
    assert rows[3]["failure_category"] == "z3_exception"
    assert rows[4]["failure_category"] == "unparseable"

    assert metrics["parseability_rate"] == pytest.approx(4 / 5)
    assert metrics["z3_execution_rate"] == pytest.approx(3 / 5)
    assert metrics["solver_verified_accuracy"] == pytest.approx(2 / 5)
    assert metrics["answer_accuracy"] == pytest.approx(2 / 5)
    assert metrics["failure_categories"] == {
        "unparseable": 1,
        "z3_exception": 1,
        "wrong_formula": 1,
        "wrong_answer": 1,
        "solver_verified_correct": 1,
    }


def test_req_bench_2959_runner_writes_required_live_repair_artifact(tmp_path: Path) -> None:
    """REQ-BENCH-2959: the runner writes required fields and manifest hash."""

    calls: list[dict[str, Any]] = []
    items = exp.selected_logic_items(max_items=8)

    def cached_pair(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(kwargs)
        return [_spec()]

    artifact = exp.run_experiment(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            output_path=tmp_path / "results" / exp.OUTPUT_FILENAME,
            max_items=8,
            started_at=10.0,
            clock=lambda: 16.0,
        ),
        cached_pair_provider=cached_pair,
        individual_model_resolver=lambda _hf_id: None,
        raw_rows_provider=lambda _items, _config: [_raw_row(item, _strict_payload(item)) for item in items],
    )
    saved = json.loads((tmp_path / "results" / exp.OUTPUT_FILENAME).read_text(encoding="utf-8"))

    assert saved == artifact
    assert calls == [{"gpu_indices": (0, 1)}]
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "live_llm_inference"
    assert artifact["z3_import_ok"] is True
    assert artifact["z3_execution_repaired"] is True
    assert artifact["n_items"] == 8
    assert artifact["parseability_rate"] == pytest.approx(1.0)
    assert artifact["z3_execution_rate"] == pytest.approx(1.0)
    assert artifact["solver_verified_accuracy"] == pytest.approx(1.0)
    assert artifact["answer_accuracy"] == pytest.approx(1.0)
    assert artifact["failure_categories"]["solver_verified_correct"] == 8
    assert artifact["formalization_manifest_sha256"] == exp.formalization_manifest_sha256(
        artifact["item_manifest"], artifact["per_item_results"]
    )
    assert artifact["duration_s"] == pytest.approx(6.0)
    assert artifact["headline_models_used"] == [MANDATED]
    assert any(row["name"] == "cached_sota_pair" and row["ok"] for row in artifact["preconditions_checked"])


def test_req_bench_2959_preconditions_and_loading_edges(tmp_path: Path) -> None:
    """REQ-BENCH-2959: preconditions and prior live rows fail closed."""

    items = exp.selected_logic_items(max_items=8)

    def missing_importer(name: str) -> object:
        if name == "z3":
            return object()
        raise ImportError(name)

    preconditions = exp.check_preconditions(
        cached_pair_provider=lambda **_: None,
        individual_model_resolver=lambda hf_id: "/tmp/model.gguf" if hf_id == MANDATED else None,
        module_importer=missing_importer,
    )
    no_source = exp.load_prior_live_rows(exp.ExperimentConfig(repo_root=tmp_path), exp.selected_logic_items())
    fallback = exp.check_preconditions(
        cached_pair_provider=lambda **_: (_ for _ in ()).throw(RuntimeError("cache down")),
        individual_model_resolver=lambda _hf_id: None,
        module_importer=lambda _name: object(),
    )
    fragment = exp.parse_or_repair_model_response(
        "{bad\n"
        + json.dumps({"formalization": items[0].formalization}, sort_keys=True)
        + "\npossible"
    )
    no_answer = exp.parse_or_repair_model_response(
        "formalization: " + json.dumps(items[0].formalization, sort_keys=True)
    )

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_text = _strict_payload(items[0])
    (raw_dir / f"{items[0].item_id}.json").write_text(raw_text, encoding="utf-8")
    source_path = tmp_path / exp.SOURCE_ARTIFACT_FILENAME
    source_path.write_text(
        json.dumps(
            {
                "models_used": [MANDATED],
                "model_attempts": [
                    {
                        "model_used": True,
                        "model_name": "Gemma4-26B-A4B-it",
                        "model_path": "/tmp/gemma.gguf",
                        "gpu_index": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prior_rows = exp.load_prior_live_rows(
        exp.ExperimentConfig(
            repo_root=tmp_path,
            source_artifact_path=source_path,
            source_raw_dir=raw_dir,
        ),
        items,
    )
    no_models_source = tmp_path / "no_models.json"
    no_models_source.write_text(json.dumps({"models_used": []}), encoding="utf-8")
    no_models_rows = exp.load_prior_live_rows(
        exp.ExperimentConfig(source_artifact_path=no_models_source, source_raw_dir=raw_dir),
        items,
    )
    no_used_attempt = exp._first_used_attempt([{"model_used": False}])  # noqa: SLF001

    assert preconditions.z3_import_ok is True
    assert preconditions.llama_cpp_import_ok is False
    assert preconditions.cached_pair_used is False
    assert preconditions.model_specs == [{**_spec(), "model_path": "/tmp/model.gguf"}]
    assert no_source == []
    assert fallback.model_specs == list(exp.MODEL_SPECS)
    assert any(row["name"] == "cached_sota_pair" and "cache down" in row["detail"] for row in fallback.rows)
    assert fragment.parseable is True
    assert fragment.answer == "possible"
    assert fragment.repair_applied is True
    assert no_answer.error == "answer_not_in_allowed_set"
    assert exp.aggregate_repair_results([])["parseability_rate"] == 0.0
    assert exp.aggregate_repair_results([{"failure_category": "unknown"}])["failure_categories"][
        "unparseable"
    ] == 1
    assert len(prior_rows) == 1
    assert prior_rows[0]["model_hf_id"] == MANDATED
    assert prior_rows[0]["output_text"] == raw_text
    assert no_models_rows == []
    assert no_used_attempt == {}
    assert (
        exp._honest_verdict(  # noqa: SLF001
            z3_import_ok=False,
            headline_models_used=[MANDATED],
            z3_execution_repaired=True,
        )
        == "blocked_z3_import_failed"
    )
    assert (
        exp._honest_verdict(  # noqa: SLF001
            z3_import_ok=True,
            headline_models_used=[],
            z3_execution_repaired=True,
        )
        == "blocked_live_sota_proposals_missing"
    )
    assert (
        exp._honest_verdict(  # noqa: SLF001
            z3_import_ok=True,
            headline_models_used=[MANDATED],
            z3_execution_repaired=False,
        )
        == "blocked_z3_execution_unrepaired"
    )
    with pytest.raises(ValueError, match="selects 8-12"):
        exp.selected_logic_items(max_items=7)
    with pytest.raises(ValueError, match="selects 8-12"):
        exp.selected_logic_items(max_items=13)
